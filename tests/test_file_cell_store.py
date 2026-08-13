"""Tests for the file-backed CellStore — the standalone vector index.

The vector-arm counterpart of ``test_sse_file_stores.py``, and it exists for the same
reason: a mantle with no object storage has to be able to answer, and giving it a local
disk must not turn the encrypted cells into readable ones.

Grouped by the question each group answers:

- ``TestPathSafety`` — can a caller-supplied id escape the cell root, or collide with another?
- ``TestCellRoundTrip`` — is it a faithful CellStore, listings included?
- ``TestCiphertextOnly`` — is anything readable on disk?
- ``TestRealQuery`` — does the vector arm return real, ranked results over a local tree?
- ``TestPersistence`` — do the cells survive the store object being thrown away and reopened?
- ``TestUnauthorizedPrincipalGetsNothing`` — the negative control.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mantle.search.mantle import cell as cell_mod
from mantle.search.mantle.engine import MantleQueryEngine
from mantle.search.mantle.file_cell_store import FileCellStore
from mantle.search.mantle.indexer import MantleIndexer
from mantle.search.mantle.oracle import GrantDenied
from mantle.search.mantle.sse.file_stores import decode_component, encode_component
from .helpers import SelfContextVerifier, make_oracle, req, self_request

DIM = 16


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _live_anchorset():
    """One anchor, so every dim-16 vector routes to one cell. Routing fan-out is
    `test_anchors`' subject; this file is about where the bytes land."""
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path / "cell-index")


@pytest.fixture
def cells(root: str) -> FileCellStore:
    return FileCellStore(root, prefix="mantle-cells")


@pytest.fixture
def oracle():
    return make_oracle()


@pytest.fixture
def indexer(oracle, cells) -> MantleIndexer:
    return MantleIndexer(oracle, cells)


@pytest.fixture
def engine(oracle, cells) -> MantleQueryEngine:
    """`cut="none"` — this file asks whether the bytes survive a round trip through a
    directory, and a result cut deciding that two of three answers do not belong together
    would answer a different question. The cut is `test_beacon_is_the_cut`'s subject."""
    return MantleQueryEngine(oracle, cells, cut="none")


def _vec(seed: int, dim: int = DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).tolist()


def _seed_corpus(indexer: MantleIndexer) -> None:
    indexer.index_artifact(
        "owner-A", "col-1",
        [
            {"artifact_id": "art-1", "chunk_id": 0, "embedding": _vec(1),
             "text": "encryption keys and cards"},
            {"artifact_id": "art-2", "chunk_id": 0, "embedding": _vec(2),
             "text": "the quick brown fox"},
        ],
        self_request("owner-A", "update"),
    )
    indexer.index_artifact(
        "owner-A", "col-2",
        [{"artifact_id": "art-3", "chunk_id": 0, "embedding": _vec(3),
          "text": "a library of lazy dogs"}],
        self_request("owner-A", "update"),
    )


def _all_files(root: str) -> list[str]:
    return [
        os.path.join(dirpath, name)
        for dirpath, _dirs, names in os.walk(root)
        for name in names
    ]


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    """A cell id is a caller-supplied string, and a filesystem reads some of those as
    instructions. The escaping law is the SSE arm's, imported rather than restated —
    these tests hold that the cell store actually applies it on all three axes."""

    @pytest.mark.parametrize("raw", [
        "owner-A", "../../etc/passwd", "a/b/c", "..", "üñî", "", "CON", "Owner-A",
    ])
    def test_every_component_axis_is_escaped_reversibly(self, raw):
        assert decode_component(encode_component(raw)) == raw

    def test_a_traversing_id_stays_inside_the_root(self, cells, root):
        cells.put("../../escaped", "../../also-escaped", b"x", "../../cluster")
        written = _all_files(root)
        assert written, "nothing was written — the assertion below would pass vacuously"
        for path in written:
            assert os.path.abspath(path).startswith(os.path.abspath(root))

    @pytest.mark.parametrize("axis", ["principal", "collection", "cluster"])
    def test_ids_differing_only_in_case_do_not_share_a_cell(self, cells, axis):
        """On a case-insensitive filesystem an unescaped id would let one owner's cell
        overwrite another's — a cross-principal data loss that looks like nothing."""
        lower = {"principal_id": "owner-a", "collection_id": "col-a", "cluster_id": "clu-a"}
        upper = dict(lower, **{f"{axis}_id": lower[f"{axis}_id"].upper()})
        cells.put(lower["principal_id"], lower["collection_id"], b"lower", lower["cluster_id"])
        cells.put(upper["principal_id"], upper["collection_id"], b"upper", upper["cluster_id"])
        assert cells.get(**lower) == b"lower"
        assert cells.get(**upper) == b"upper"


# ---------------------------------------------------------------------------
# CellStore Protocol
# ---------------------------------------------------------------------------


class TestCellRoundTrip:
    def test_missing_returns_none(self, cells):
        assert cells.get("owner-A", "col-1", "clu-1") is None

    def test_put_get_overwrite_delete(self, cells):
        cells.put("owner-A", "col-1", b"one", "clu-1")
        assert cells.get("owner-A", "col-1", "clu-1") == b"one"
        cells.put("owner-A", "col-1", b"two", "clu-1")
        assert cells.get("owner-A", "col-1", "clu-1") == b"two"
        cells.delete("owner-A", "col-1", "clu-1")
        assert cells.get("owner-A", "col-1", "clu-1") is None

    def test_delete_missing_is_a_noop(self, cells):
        cells.delete("owner-A", "col-1", "clu-1")     # must not raise

    def test_owner_isolation(self, cells):
        cells.put("owner-A", "col-1", b"A", "clu-1")
        cells.put("owner-B", "col-1", b"B", "clu-1")
        assert cells.get("owner-A", "col-1", "clu-1") == b"A"
        assert cells.get("owner-B", "col-1", "clu-1") == b"B"

    def test_clusters_of_one_collection_are_separate_cells(self, cells):
        cells.put("owner-A", "col-1", b"zero", "anchor-0")
        cells.put("owner-A", "col-1", b"one", "anchor-1")
        assert cells.get("owner-A", "col-1", "anchor-0") == b"zero"
        assert sorted(cells.list_clusters("owner-A", "col-1")) == ["anchor-0", "anchor-1"]

    def test_list_cells_names_the_collections(self, cells):
        cells.put("owner-A", "col-1", b"x", "anchor-0")
        cells.put("owner-A", "col-2", b"y", "anchor-0")
        cells.put("owner-B", "col-3", b"z", "anchor-0")
        assert sorted(cells.list_cells("owner-A")) == ["col-1", "col-2"]
        assert cells.list_cells("owner-B") == ["col-3"]
        assert cells.list_cells("owner-nobody") == []

    def test_listings_ignore_non_index_files(self, cells, root):
        """`mkstemp` leftovers from an interrupted write must not be read back as cluster
        ids, or an admin purge would iterate over names that address nothing."""
        cells.put("owner-A", "col-1", b"x", "anchor-0")
        leftover = os.path.join(
            os.path.dirname(cells._cell_path("owner-A", "col-1", "anchor-0")), "tmpXYZ",
        )
        with open(leftover, "wb") as fh:
            fh.write(b"junk")
        assert cells.list_clusters("owner-A", "col-1") == ["anchor-0"]

    def test_rejects_non_bytes(self, cells):
        with pytest.raises(TypeError):
            cells.put("owner-A", "col-1", "not bytes", "clu-1")  # type: ignore[arg-type]

    def test_requires_a_root(self):
        """An empty root would put the cells in the process's working directory — a location
        nobody chose and nobody would find again."""
        with pytest.raises(ValueError):
            FileCellStore("")

    def test_segments_do_not_share_a_tree(self, root):
        committed = FileCellStore(root, prefix="mantle-cells")
        draft = FileCellStore(root, prefix="mantle-cells-draft")
        committed.put("owner-A", "col-1", b"committed", "anchor-0")
        draft.put("owner-A", "col-1", b"draft", "anchor-0")
        assert committed.get("owner-A", "col-1", "anchor-0") == b"committed"
        assert draft.get("owner-A", "col-1", "anchor-0") == b"draft"


# ---------------------------------------------------------------------------
# Encrypted round-trip + what lands on disk
# ---------------------------------------------------------------------------


class TestCiphertextOnly:
    def test_encrypted_blobs_round_trip_byte_for_byte(self, cells):
        """The store must not re-encode what it stores. Any transformation — a newline
        translation, a text-mode open, a trailing byte — breaks GCM authentication on read,
        and the arm would report tampering it caused itself."""
        key = bytes(range(32))
        aad = cell_mod.cell_aad("col-1", "anchor-0")
        chunks = [{"artifact_id": "art-1", "chunk_id": 0, "embedding": _vec(1)}]
        blob = cell_mod.pack_cell(chunks, key, collection_id=aad)
        cells.put("owner-A", "col-1", blob, "anchor-0")
        assert cells.get("owner-A", "col-1", "anchor-0") == blob
        assert cell_mod.unpack_cell(
            cells.get("owner-A", "col-1", "anchor-0"), key, collection_id=aad,
        ) == chunks

    def test_a_cell_does_not_open_under_another_slot_s_aad(self, cells):
        """The envelope is unweakened by the move to disk: the blob is still bound to its
        `collection:cluster` slot, so a cell copied to another cluster's path fails to open."""
        key = bytes(range(32))
        blob = cell_mod.pack_cell(
            [{"artifact_id": "art-1", "chunk_id": 0, "embedding": _vec(1)}],
            key, collection_id=cell_mod.cell_aad("col-1", "anchor-0"),
        )
        cells.put("owner-A", "col-1", blob, "anchor-1")      # same bytes, wrong slot
        with pytest.raises(cell_mod.CellTampered):
            cell_mod.unpack_cell(
                cells.get("owner-A", "col-1", "anchor-1"), key,
                collection_id=cell_mod.cell_aad("col-1", "anchor-1"),
            )

    def test_nothing_readable_reaches_the_disk(self, indexer, root):
        """The invariant that makes a file-backed cell tree acceptable at all. The vector arm
        stores chunk TEXT alongside the embedding, so a local store that wrote anything in the
        clear would put the artifact's content on disk in plaintext."""
        _seed_corpus(indexer)
        files = _all_files(root)
        assert files, "nothing was indexed — the scan below would pass vacuously"
        corpus = b"".join(open(p, "rb").read() for p in files)
        for secret in (
            b"encryption", b"keys", b"cards", b"quick", b"brown", b"fox", b"library",
            b"lazy", b"dogs", b"art-1", b"art-2", b"art-3", b"col-1", b"col-2", b"owner-A",
        ):
            assert secret not in corpus, f"{secret!r} is READABLE in the local cell tree"

    def test_no_readable_side_car_is_written(self, indexer, root):
        """No convenience file — a listing of owners, a `.json` of clusters — beside the
        ciphertext. Such a file would be exactly the plaintext the encryption exists to remove,
        and it would look like harmless bookkeeping."""
        _seed_corpus(indexer)
        files = _all_files(root)
        assert files, "nothing was indexed — the scan below would pass vacuously"
        for path in files:
            assert path.endswith(".cell"), f"non-ciphertext file in the cell tree: {path}"


# ---------------------------------------------------------------------------
# A real query over a file-backed cell tree
# ---------------------------------------------------------------------------


class TestRealQuery:
    def test_query_returns_ranked_results(self, indexer, engine):
        """A faithful dictionary store is not enough on its own: the engine must read back
        cells it can decrypt and score, or the vector arm answers empty forever —
        indistinguishable from an unindexed corpus."""
        _seed_corpus(indexer)
        hits = engine.search(
            _vec(1), [("owner-A", "col-1"), ("owner-A", "col-2")], req(),
        )
        assert {h.artifact_id for h in hits} == {"art-1", "art-2", "art-3"}
        # The query IS art-1's vector, so art-1 ranks first at cosine 1.
        assert hits[0].artifact_id == "art-1"
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_query_respects_the_authorized_collection_set(self, indexer, engine):
        _seed_corpus(indexer)
        hits = engine.search(_vec(1), [("owner-A", "col-1")], req())
        assert {h.artifact_id for h in hits} == {"art-1", "art-2"}

    def test_removal_actually_removes(self, indexer, engine):
        """If `delete`/rewrite left the cell behind, a removed artifact would keep matching —
        the index would only ever grow and would answer for content that no longer exists."""
        _seed_corpus(indexer)
        indexer.remove_artifact(
            "owner-A", "col-1", "art-1", self_request("owner-A", "update"),
        )
        engine.evict_cache()
        hits = engine.search(_vec(1), [("owner-A", "col-1")], req())
        assert {h.artifact_id for h in hits} == {"art-2"}


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_cells_survive_a_reopen(self, oracle, root):
        """The one thing the in-memory store cannot do, and the reason this backend exists.
        The store object is discarded and rebuilt over the same directory, exactly as a mantle
        restart does."""
        writer = FileCellStore(root, prefix="mantle-cells")
        _seed_corpus(MantleIndexer(oracle, writer))
        del writer

        reopened = MantleQueryEngine(
            oracle, FileCellStore(root, prefix="mantle-cells"), cut="none",
        )
        hits = reopened.search(_vec(1), [("owner-A", "col-1")], req())
        assert {h.artifact_id for h in hits} == {"art-1", "art-2"}


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


class TestUnauthorizedPrincipalGetsNothing:
    """The local disk is another untrusted server. Custody is unchanged by where the bytes
    live, so a principal with no grant gets no key and therefore no cells."""

    def test_no_key_is_issued(self, root):
        from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
        from cryptography.fernet import Fernet
        from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
        from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

        oracle = OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())),
            grant_verifier=SelfContextVerifier(),
        )
        _seed_corpus(MantleIndexer(oracle, FileCellStore(root)))

        set_acting_principal(ActingPrincipal(
            principal_id="mallory", principal_type="user", source="test",
        ))
        outsider = KeyRequest(
            requester_id="mallory", purpose=KeyPurpose.GRANT,
            requester_type="user", action="read",
        )
        with pytest.raises(GrantDenied):
            oracle.derive_cell_key("owner-A", "col-1", "anchor-0", outsider)

    def test_the_query_engine_yields_no_hits_even_with_forged_contexts(self, root):
        from cryptography.fernet import Fernet
        from mantle.search.mantle.oracle import (
            FernetMasterKeyStore, KeyPurpose, KeyRequest, OracleService,
        )
        from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

        oracle = OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())),
            grant_verifier=SelfContextVerifier(),
        )
        store = FileCellStore(root)
        _seed_corpus(MantleIndexer(oracle, store))

        set_acting_principal(ActingPrincipal(
            principal_id="mallory", principal_type="user", source="test",
        ))
        outsider = KeyRequest(
            requester_id="mallory", purpose=KeyPurpose.GRANT,
            requester_type="user", action="read",
        )
        with pytest.raises(GrantDenied):
            MantleQueryEngine(oracle, store).search(
                _vec(1), [("owner-A", "col-1")], outsider,
            )
