"""`/.well-known/agience-source` — what this instance runs, and where to verifiably get it.

The point is the join, not either half: `/git` on its own serves source a node happens to hold;
`/version` on its own names a revision with nowhere to get it. Together they answer the question
that matters to someone forking from a node — is this the code the instance is actually running? —
and they answer it without a signature, because a git object id is already a hash of the content.

    git clone https://<host>/git/prism && git -C prism cat-file -e <revision>^{commit}

It attests the build, not the process. Nothing served over HTTP by a process can prove that
process is unmodified. The check that catches a lying node is the deploy's, comparing the running
image digest against what the repo records, and it is made from outside the box. These tests assert
the join and the honesty of the reporting; they do not pretend to more.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mantle.main import app
import mantle.main as main_mod
from mantle.routers import git_router as git_mod

client = TestClient(app)

SHA = "b" * 40


@pytest.fixture
def built_from(monkeypatch):
    """This image was built from mantle and prism, at known revisions."""
    monkeypatch.setattr(main_mod, "BUILD_INFO", {
        "version": SHA, "build_time": "2026-08-24T00:00:00Z",
        "components": {"agience-mantle": SHA, "agience-prism-py": "c" * 40},
    })


def test_a_flat_build_arg_widens_into_a_revision_set():
    """A build ARG is a string; the set is parsed once, where the stamp is read."""
    assert main_mod._parse_components("a=1,b=2") == {"a": "1", "b": "2"}
    assert main_mod._parse_components(" a=1 , b=2 ") == {"a": "1", "b": "2"}


@pytest.mark.parametrize("raw", ["", None, "garbage", "a=", "=1", ",,,"])
def test_an_unparseable_pair_is_dropped_rather_than_guessed(raw):
    """A wrong SHA is worse than a missing one. A component whose revision cannot be read must be
    absent from the attestation — present with a bad value, it would send someone to verify a clone
    against a commit that does not exist and read the failure as a compromised node."""
    assert main_mod._parse_components(raw) == {}


def test_an_unstamped_image_reports_no_components_rather_than_failing(monkeypatch):
    """Older images carry no `components` key. That is "this build did not say", and it must not be
    an error — the endpoint still has to answer for a node running a pre-stamp image."""
    monkeypatch.setattr(main_mod, "BUILD_INFO", {"version": "x", "build_time": "", "components": {}})
    r = client.get("/.well-known/agience-source")
    assert r.status_code == 200
    assert r.json()["components"] == []


def test_a_served_component_carries_a_url_and_a_runnable_verify(built_from, monkeypatch, tmp_path):
    """prism is in `SERVED`, so the offer joins its revision to a clone URL on THIS host."""
    monkeypatch.setenv("MANTLE_GIT_ROOT", str(tmp_path))
    body = client.get("/.well-known/agience-source").json()
    prism = next(c for c in body["components"] if c["repo"] == "agience-prism-py")

    assert prism["served_here"] is True
    assert prism["source_url"].endswith("/git/prism")
    assert prism["license"] == "Apache-2.0"
    # The verify line is the whole product: a command a reader can paste, naming this node's URL and
    # the exact commit. `^{commit}` is what makes `cat-file -e` a commit check rather than a
    # "some object with this id" check.
    assert prism["revision"] in prism["verify"]
    assert "cat-file -e" in prism["verify"] and "^{commit}" in prism["verify"]


def test_mantle_is_now_served_alongside_the_instruments(built_from, monkeypatch, tmp_path):
    """The scope, asserted end to end.

    mantle is in `SERVED` because its licence is measured: it is Apache-2.0, so the objection that
    produced the original instruments-only scope never applied to it. A node therefore offers the
    source of the service you are talking to, with the revision it is running.
    """
    monkeypatch.setenv("MANTLE_GIT_ROOT", str(tmp_path))
    body = client.get("/.well-known/agience-source").json()
    mantle = next(c for c in body["components"] if c["repo"] == "agience-mantle")

    assert mantle["served_here"] is True
    assert mantle["source_url"].endswith("/git/mantle")
    assert mantle["license"] == "Apache-2.0"
    assert mantle["revision"] == SHA


def test_a_component_whose_source_is_not_served_says_so_instead_of_vanishing(monkeypatch, tmp_path):
    """The honesty requirement. A component in the image whose source this node does not carry is
    listed with `served_here: false` and no URL — never omitted, because an omission reads as "there
    is nothing else in this image".

    `agience-origin` stands in for that class here: AGPL-3.0-only and deliberately outside `SERVED`.
    """
    monkeypatch.setattr(main_mod, "BUILD_INFO", {
        "version": SHA, "build_time": "",
        "components": {"agience-mantle": SHA, "agience-origin": "d" * 40},
    })
    monkeypatch.setenv("MANTLE_GIT_ROOT", str(tmp_path))
    body = client.get("/.well-known/agience-source").json()
    origin = next(c for c in body["components"] if c["repo"] == "agience-origin")

    assert origin["served_here"] is False
    assert "source_url" not in origin
    assert "does not serve" in origin["note"]
    assert origin["license"] == "AGPL-3.0-only"
    assert origin["revision"] == "d" * 40     # still attested, even though not served


def test_with_the_git_surface_off_nothing_claims_to_be_served(built_from, monkeypatch):
    """A node with `/git` off still attests what it runs — it just has no source to offer, and the
    reason names the surface rather than the component."""
    monkeypatch.delenv("MANTLE_GIT_ROOT", raising=False)
    body = client.get("/.well-known/agience-source").json()
    assert body["git_surface"] == "off"
    assert all(c["served_here"] is False for c in body["components"])
    assert any("git surface is off" in c.get("note", "") for c in body["components"])


def test_mantle_is_reported_as_apache_not_agpl():
    """The intuitive mistake, pinned. mantle is "the platform" by role and Apache-2.0 by
    licence — measured from its LICENSE and pyproject, 2026-08-24 — and that decides who owes a
    source offer to whom: serving `mantle.agience.ai` carries no network-copyleft requirement.
    """
    assert main_mod.COMPONENT_LICENSES["agience-mantle"] == "Apache-2.0"
    for agpl in ("agience-origin", "agience-chorus", "agience-ember", "agience-observe"):
        assert main_mod.COMPONENT_LICENSES[agpl] == "AGPL-3.0-only"


def test_every_served_repo_has_a_declared_licence():
    """`git_router.SERVED` and the licence table are two lists that must not drift: a repo this node
    publishes with no licence recorded would be offered with `"unknown"` beside it."""
    for name, spec in git_mod.SERVED.items():
        assert spec["repo"] in main_mod.COMPONENT_LICENSES, \
            "%s (%s) is served with no licence declared" % (name, spec["repo"])
        assert spec["license"] == main_mod.COMPONENT_LICENSES[spec["repo"]], \
            "%s: git_router says %s, COMPONENT_LICENSES says %s" % (
                name, spec["license"], main_mod.COMPONENT_LICENSES[spec["repo"]])


def test_the_agpl_notice_is_addressed_to_the_operator_who_needs_it(built_from):
    """AGPL-3.0 §13 binds LICENSEES. A copyright holder running its own AGPL code owes nothing to
    itself; the obligation lands on a third party who modifies a component and serves it over a
    network — which is exactly the fork-and-run story this surface enables. The notice ships with
    the node so a downstream operator has a mechanism rather than a homework assignment."""
    body = client.get("/.well-known/agience-source").json()
    notice = body["agpl_notice"]
    assert "§13" in notice and "MODIFIES" in notice
    assert "Apache-2.0 carry no such requirement" in notice
