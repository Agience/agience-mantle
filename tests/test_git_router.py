"""`/git` — a node serves its own source, read-only, and only what it decided to serve.

The claim worth testing is not "the endpoint returns 200". It is that **a real git client can clone
from it** and that **nothing outside the allowlist is reachable**. The first is asserted by running
`git clone` against the app; the second by trying the things an attacker or a mistake would try.

These run as an authenticated caller (`conftest.override_dependencies`). The router requires
auth, and anonymous pull — what would make a node a public mirror anyone can fork from — is a
further deliberate act that is not enabled. Nothing here says otherwise.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from mantle.main import app
from mantle.routers import git_router as mod

client = TestClient(app)

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed on this machine")


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A real bare repo with one commit, published under the `prism` name.

    A real repository rather than a fixture directory: this surface is `git upload-pack` and the
    thing being asserted is that git itself is satisfied, which only a real object database can
    answer.
    """
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run([GIT, "init", "-q", "-b", "main", str(work)], check=True)
    (work / "README.md").write_text("the instrument\n", encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@e", "PATH": __import__("os").environ.get("PATH", "")}
    subprocess.run([GIT, "-C", str(work), "add", "-A"], check=True, env=env)
    subprocess.run([GIT, "-C", str(work), "commit", "-qm", "one"], check=True, env=env)

    root = tmp_path / "bares"
    root.mkdir()
    bare = root / "agience-prism-py.git"
    subprocess.run([GIT, "clone", "-q", "--bare", str(work), str(bare)], check=True, env=env)

    monkeypatch.setenv("MANTLE_GIT_ROOT", str(root))
    return root


def test_the_surface_is_off_until_it_is_configured(monkeypatch):
    """A node does not serve its source by default.

    Turning it on is a deliberate act on a node whose reachability its operator has thought about
    (`NODE-DEPLOYMENT-AND-FLEET.md` §4, "the apex question"). The refusal names the variable, so an
    operator who expected it on learns why it is not.
    """
    monkeypatch.delenv("MANTLE_GIT_ROOT", raising=False)
    r = client.get("/git/prism/info/refs", params={"service": "git-upload-pack"})
    assert r.status_code == 404
    assert r.json()["error"] == "git_surface_not_configured"
    assert "MANTLE_GIT_ROOT" in r.json()["message"]


def test_the_advertisement_is_what_a_git_client_expects(served):
    """The pkt-line header is computed, not hardcoded — a wrong length looks right in a terminal and
    is rejected by every client."""
    r = client.get("/git/prism/info/refs", params={"service": "git-upload-pack"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-git-upload-pack-advertisement")
    body = r.content
    assert body.startswith(b"001e# service=git-upload-pack\n0000"), body[:60]
    assert b"refs/heads/main" in body


def test_a_repository_this_node_does_not_publish_is_404(served):
    """The allowlist is the licence boundary, and being on disk must not be what decides.

    `agience-mantle` is not the right subject here: mantle is in `SERVED` (measured Apache-2.0, so
    the instruments-only objection never applied to it), but under the clone name `mantle`, so a
    request for `agience-mantle` 404s as a name miss rather than as the allowlist doing its job. A
    test that passes for a reason it does not state is worse than one that fails.

    `agience-origin` is the honest subject: AGPL-3.0-only, deliberately outside `SERVED`, and
    publishing it is a separate decision with a different consequence.
    """
    (served / "agience-origin.git").mkdir()
    (served / "agience-origin.git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (served / "agience-origin.git" / "objects").mkdir()
    r = client.get("/git/agience-origin/info/refs", params={"service": "git-upload-pack"})
    assert r.status_code == 404, "a repo on disk but not in SERVED was reachable"

    # And the positive control: a name that IS in the allowlist resolves, so the 404 above is the
    # allowlist refusing rather than the fixture being empty.
    ok = client.get("/git/prism/info/refs", params={"service": "git-upload-pack"})
    assert ok.status_code == 200, "the allowlist refused a repo it should serve — fixture is broken"


@pytest.mark.parametrize("name", [
    "../../etc/passwd", "..", "../bares", "prism/../../x", "PRISM", "pri sm",
])
def test_a_name_that_is_not_a_name_is_refused(served, name):
    """Traversal and lookalikes, rejected by the pattern before anything touches the filesystem.

    `PRISM` is in the list because a case-insensitive filesystem would otherwise let the
    allowlist be bypassed by capitalisation — the pattern is lowercase-only for that reason.
    """
    r = client.get("/git/%s/info/refs" % name, params={"service": "git-upload-pack"})
    assert r.status_code in (404, 400, 405), "%r reached something: %s" % (name, r.status_code)
    assert b"root:" not in r.content


def test_push_is_refused_by_name_not_by_absence(served):
    """403 with a reason, not 404.

    A client that tried to push should learn this is a read-only mirror, not that the repository
    does not exist — those lead to different next actions. Declared as a route so the refusal is a
    property of the table rather than an accident of what is not mounted.
    """
    r = client.post("/git/prism/git-receive-pack", content=b"")
    assert r.status_code == 403
    assert "read-only" in r.text

    adv = client.get("/git/prism/info/refs", params={"service": "git-receive-pack"})
    assert adv.status_code == 403
    assert "read-only" in adv.text


def test_the_router_offers_no_write_surface():
    """Asserted against the app's own route table, so adding one is a deliberate, visible change."""
    writes = {(m, r.path) for r in app.routes
              for m in (getattr(r, "methods", None) or ())
              if str(getattr(r, "path", "")).startswith("/git")
              and m in {"PUT", "PATCH", "DELETE"}}
    assert not writes, "the /git surface has acquired write routes: %s" % sorted(writes)


def test_a_real_git_client_can_clone_from_this_node(served, tmp_path):
    """The one that matters. Everything above checks a response shape; this checks that `git`
    itself is satisfied — negotiation, packfile, refs and all.

    Served over a real socket because `git` is a separate process and cannot speak to a TestClient.
    The server runs in a thread on an ephemeral port for the duration of the test.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = __import__("time").time() + 30
        while not server.started and __import__("time").time() < deadline:
            __import__("time").sleep(0.05)
        assert server.started, "the test server did not come up"
        port = server.servers[0].sockets[0].getsockname()[1]

        dest = tmp_path / "cloned"
        proc = subprocess.run(
            [GIT, "clone", "-q", "http://127.0.0.1:%d/git/prism" % port, str(dest)],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, "git clone failed:\n%s" % proc.stderr
        assert (dest / "README.md").read_text(encoding="utf-8") == "the instrument\n"
    finally:
        server.should_exit = True
        thread.join(timeout=30)
