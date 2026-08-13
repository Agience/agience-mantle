from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mantle.services import collection_service as svc


@patch("mantle.services.collection_service.db_get_current_in_any_collection_many")
@patch("mantle.services.collection_service.db_get_active_collection_ids_for_user", return_value=["granted-col"])
@patch("mantle.services.collection_service.db_get_collections_by_owner_id", return_value=[])
def test_get_collection_cards_batch_global_includes_grant_accessible_collections(
    _mock_owned,
    _mock_grants,
    mock_current_in_any,
):
    expected = SimpleNamespace(root_id="host-root", id="host-version", is_archived=False)

    def side_effect(_db, root_ids, collection_ids):
        if "host-root" in root_ids and "granted-col" in collection_ids:
            return {"host-root": expected}
        return {}

    mock_current_in_any.side_effect = side_effect

    artifacts = svc.get_collection_artifacts_batch_global(MagicMock(), "user-1", ["host-root"])

    assert artifacts == [expected]
    mock_current_in_any.assert_any_call(AnyDbMatcher(), ["host-root"], ["granted-col"])


@patch("mantle.services.collection_service.db_get_current_in_any_collection_many", return_value={})
@patch("mantle.services.collection_service.db_get_active_collection_ids_for_user",
       return_value=["col-b", "col-c", "col-d"])
@patch("mantle.services.collection_service.db_get_collections_by_owner_id",
       return_value=[SimpleNamespace(id="col-a")])
def test_batch_global_costs_one_lookup_for_the_whole_set(
    _mock_owned,
    _mock_grants,
    mock_current_in_any,
):
    """The "batch" in the name has to mean something.

    Asking root by root re-reads a lineage per root, and asking each collection separately makes
    it O(roots x collections) round trips — for an answer that depends on one lineage read over
    the whole set. Two roots across four accessible collections is ONE lookup, not two and not
    eight."""
    svc.get_collection_artifacts_batch_global(MagicMock(), "user-1", ["root-1", "root-2"])

    assert mock_current_in_any.call_count == 1
    call = mock_current_in_any.call_args
    assert call.args[1] == ["root-1", "root-2"]
    # Owned collections lead, then grant-reachable ones — a stable order, so a root visible
    # through more than one collection resolves the same way on every call.
    assert call.args[2] == ["col-a", "col-b", "col-c", "col-d"]


@patch("mantle.services.collection_service.db_get_current_in_any_collection_many", return_value={})
@patch("mantle.services.collection_service.db_get_active_collection_ids_for_user",
       return_value=["col-a", "col-b"])
@patch("mantle.services.collection_service.db_get_collections_by_owner_id",
       return_value=[SimpleNamespace(id="col-a")])
def test_batch_global_does_not_ask_about_the_same_collection_twice(
    _mock_owned,
    _mock_grants,
    mock_current_in_any,
):
    """A collection can be both owned and granted; it is still one collection."""
    svc.get_collection_artifacts_batch_global(MagicMock(), "user-1", ["root-1"])

    assert mock_current_in_any.call_args.args[2] == ["col-a", "col-b"]


@patch("mantle.services.collection_service.db_get_current_in_any_collection_many", return_value={})
@patch("mantle.services.collection_service.db_get_active_collection_ids_for_user", return_value=[])
@patch("mantle.services.collection_service.db_get_collections_by_owner_id", return_value=[])
def test_batch_global_survives_a_grant_lookup_that_raises(
    _mock_owned,
    mock_grants,
    _mock_current_in_any,
):
    """A failed grant resolution degrades to the owned set; it does not take the request down."""
    mock_grants.side_effect = RuntimeError("grant plane unavailable")

    assert svc.get_collection_artifacts_batch_global(MagicMock(), "user-1", ["root-1"]) == []


@patch("mantle.services.collection_service._attach_committed_collection_ids")
@patch("mantle.services.collection_service.db_get_current_in_collection_many")
def test_by_root_ids_reads_the_whole_set_once_and_keeps_the_caller_order(
    mock_current_many,
    _mock_attach,
):
    """One store call for the page, results in the order the roots were named, repeats collapsed.

    A root the collection does not hold is skipped rather than yielded as a hole, so the result
    can be shorter than the input — which is why it is keyed back through the caller's own list
    instead of being read off the mapping's iteration order."""
    a1 = SimpleNamespace(root_id="root-1", id="v1", is_archived=False)
    a3 = SimpleNamespace(root_id="root-3", id="v3", is_archived=False)
    mock_current_many.return_value = {"root-3": a3, "root-1": a1}

    got = svc.get_collection_artifacts_by_root_ids(
        MagicMock(), "user-1", "col-1", ["root-1", "root-2", "root-3", "root-1"])

    assert got == [a1, a3]
    assert mock_current_many.call_count == 1
    assert mock_current_many.call_args.args[1] == "col-1"
    assert mock_current_many.call_args.args[2] == ["root-1", "root-2", "root-3"]


@patch("mantle.services.collection_service._attach_committed_collection_ids")
@patch("mantle.services.collection_service.db_get_current_in_collection_many")
def test_by_root_ids_omits_archived_versions(mock_current_many, _mock_attach):
    """An archived version is not a current member, whatever the lineage read returned."""
    live = SimpleNamespace(root_id="root-1", id="v1", is_archived=False)
    dead = SimpleNamespace(root_id="root-2", id="v2", is_archived=True)
    mock_current_many.return_value = {"root-1": live, "root-2": dead}

    assert svc.get_collection_artifacts_by_root_ids(
        MagicMock(), "user-1", "col-1", ["root-1", "root-2"]) == [live]


class AnyDbMatcher:
    def __eq__(self, other):
        return True
