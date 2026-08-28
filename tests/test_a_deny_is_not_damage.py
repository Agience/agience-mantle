"""A refused document and a broken one are different answers, and `query_documents` says which.

`unreadable` means the bytes will not open. `denied` means key custody was REFUSED — the caller
may not read this document. `propagate='[]'` on a `contains` edge is `attenuation`'s absorbing
deny, so an artifact behind one is working exactly as configured, and filing it under `unreadable`
would report a correctly-provisioned store as a damaged one.

Measured on 71/home, which is why these tests exist: 6 such edges out of 2,158,434 — one of them
over a real collection ("Mantle work products") rather than a test fixture — and a single
`GrantDenied` escaping the hydration loop ended a rebuild of all 2,165,867 artifacts. `GrantDenied`
is `KeyCustodyDenied` -> `PermissionError` and is NOT a `ContentDecryptionError`, so the existing
`unreadable="skip"` clause never saw it.
"""
import pytest

from mantle.db import lattice_api as la
from mantle.db.backend import COLLECTION_ARTIFACTS
from mantle.services.acting_principal import NoActingPrincipal

from mantle.search.mantle.oracle import GrantDenied


class _Artifacts:
    def __init__(self, docs):
        self._docs = docs

    def list_artifacts(self, **_kw):
        return list(self._docs)


class _DB:
    def __init__(self, docs):
        self.artifacts = _Artifacts(docs)


def _docs(n):
    return [{"id": f"a{i}", "content_type": "text/markdown"} for i in range(n)]


def _hydrate_raising_on(ids, exc_factory):
    """A `from_lattice_doc` that refuses exactly the named ids and hydrates the rest."""
    def _fn(d, _cls):
        if d["id"] in ids:
            raise exc_factory(d["id"])
        return d
    return _fn


def _grant_denied(which):
    return GrantDenied(f"'writer' holds no 'read' grant reaching '{which}'")


def test_a_denial_ends_the_scan_by_default(monkeypatch):
    """`KeyCustodyDenied`'s own rule: "not authorized" must not become "no results"."""
    monkeypatch.setattr(la, "from_lattice_doc", _hydrate_raising_on({"a1"}, _grant_denied))
    with pytest.raises(GrantDenied):
        la.query_documents(_DB(_docs(3)), dict, COLLECTION_ARTIFACTS, {})


def test_omit_leaves_the_refused_out_and_names_it(monkeypatch):
    monkeypatch.setattr(la, "from_lattice_doc", _hydrate_raising_on({"a1"}, _grant_denied))
    refused: list = []
    skipped: list = []
    out = la.query_documents(_DB(_docs(3)), dict, COLLECTION_ARTIFACTS, {},
                             denied="omit", denied_out=refused,
                             unreadable="skip", skipped_out=skipped)
    assert [d["id"] for d in out] == ["a0", "a2"]
    assert [r[0] for r in refused] == ["a1"]
    # The whole point: a deny is NOT filed as damage.
    assert skipped == []
    assert "GrantDenied" in refused[0][1]


def test_being_denied_everything_still_raises(monkeypatch):
    """A misprovisioned run, not a store with a few deny edges in it.

    Returning [] here would report "nothing to do" for "I was allowed to see nothing" — the same
    failure the `NoActingPrincipal` guard prevents one layer in.
    """
    monkeypatch.setattr(la, "from_lattice_doc",
                        _hydrate_raising_on({"a0", "a1", "a2"}, _grant_denied))
    with pytest.raises(PermissionError):
        la.query_documents(_DB(_docs(3)), dict, COLLECTION_ARTIFACTS, {},
                           denied="omit", denied_out=[])


def test_no_acting_principal_is_never_omitted(monkeypatch):
    """It subclasses the same base and fails for EVERY document, so it must not be omitted."""
    monkeypatch.setattr(
        la, "from_lattice_doc",
        _hydrate_raising_on({"a1"}, lambda w: NoActingPrincipal("no caller in scope")))
    with pytest.raises(NoActingPrincipal):
        la.query_documents(_DB(_docs(3)), dict, COLLECTION_ARTIFACTS, {},
                           denied="omit", denied_out=[])


def test_damage_is_still_damage(monkeypatch):
    """`unreadable` keeps its meaning: a document that will not decrypt is skipped, not refused."""
    def _boom(which):
        return la._boundary.ContentDecryptionError(f"{which}: content will not open")

    monkeypatch.setattr(la, "from_lattice_doc", _hydrate_raising_on({"a1"}, _boom))
    refused: list = []
    skipped: list = []
    out = la.query_documents(_DB(_docs(3)), dict, COLLECTION_ARTIFACTS, {},
                             unreadable="skip", skipped_out=skipped,
                             denied="omit", denied_out=refused)
    assert [d["id"] for d in out] == ["a0", "a2"]
    assert [s[0] for s in skipped] == ["a1"]
    assert refused == []


def test_the_modes_are_validated():
    with pytest.raises(ValueError, match="denied must be"):
        la.query_documents(_DB(_docs(1)), dict, COLLECTION_ARTIFACTS, {}, denied="omitt")
